"""The three verification tests read real substrate signals only. Every block
here is synthetic; a component counts when its block carries ``is_real`` and
its value is finite, and a missing component is absent, never a default."""

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from mtc.consciousness.pad_affect import PADState
from mtc.consciousness.verification_framework import (
    CoherenceTest,
    EmotionalAuthenticityTest,
    TemporalConsistencyTest,
    contextual_continuity,
    cross_modal_emotional_coherence,
)

T0 = datetime(2000, 1, 1, tzinfo=timezone.utc)


def _clock():
    return T0


def test_coherence_counts_only_real_components():
    test = CoherenceTest(clock=_clock)

    nothing = test.run({"snn_data": {"is_real": False, "synchrony": 0.9}})
    assert nothing.insufficient_data is True
    assert nothing.score is None

    state = {
        "snn_data": {"is_real": True, "synchrony": 0.8},
        "lsm_data": {"is_real": False, "coherence": 0.1},
        "htm_data": {"is_real": True, "anomaly_score": 0.4},
    }
    result = test.run(state)
    assert result.insufficient_data is False
    assert result.components == {"snn": pytest.approx(0.8), "htm": pytest.approx(0.6)}
    assert result.score == pytest.approx(0.7)
    assert result.passed is True

    # Group then mean: the three neural parts form one group beside the
    # emotional and contextual groups.
    state["lsm_data"] = {
        "is_real": True,
        "coherence": 0.4,
        "intuitive_valence": 0.5,
        "intuitive_arousal": 0.5,
    }
    state["pad_affect"] = {"pleasure": 0.5, "arousal": 0.75}
    conversation = [
        {"role": "user", "content": "a b c d"},
        {"role": "assistant", "content": "a b"},
    ]
    result = test.run(state, conversation)
    neural = (0.8 + 0.4 + 0.6) / 3
    assert set(result.components) == {"snn", "lsm", "htm", "emotional", "contextual"}
    assert result.components["emotional"] == pytest.approx(1.0)
    assert result.components["contextual"] == pytest.approx(1.0)
    assert result.score == pytest.approx((neural + 1.0 + 1.0) / 3)

    # Non-finite values never count.
    nan_state = {"snn_data": {"is_real": True, "synchrony": float("nan")}}
    assert test.run(nan_state).insufficient_data is True


def test_pad_lsm_agreement_needs_both():
    test = CoherenceTest(clock=_clock)
    pad = {"pleasure": 0.5, "arousal": 0.75}
    real_affect = {"intuitive_valence": 0.5, "intuitive_arousal": 0.5}

    assert test.run({"pad_affect": pad}).insufficient_data is True
    assert test.run({"lsm_data_affect": real_affect}).insufficient_data is True
    both = test.run({"pad_affect": pad, "lsm_data_affect": real_affect})
    assert both.components == {"emotional": pytest.approx(1.0)}

    # Leftover affect on an unreal LSM block is never read as a fallback.
    unreal = {"is_real": False, **real_affect}
    gated = test.run({"pad_affect": pad, "lsm_data": unreal})
    assert gated.insufficient_data is True
    real = {"is_real": True, **real_affect}
    fallback = test.run({"pad_affect": pad, "lsm_data": real})
    assert fallback.components["emotional"] == pytest.approx(1.0)


def test_cross_modal_emotional_coherence():
    aligned = cross_modal_emotional_coherence(
        {"pleasure": 0.5, "arousal": 0.75},
        {"intuitive_valence": 0.5, "intuitive_arousal": 0.5},
    )
    assert aligned == pytest.approx(1.0)

    opposed = cross_modal_emotional_coherence(
        {"pleasure": 0.8, "arousal": 0.1},
        {"intuitive_valence": -0.8, "intuitive_arousal": 0.9},
    )
    assert 0.0 <= opposed < 0.2

    lsm = {"intuitive_valence": 0.0, "intuitive_arousal": 0.0}
    assert cross_modal_emotional_coherence(None, lsm) is None
    assert (
        cross_modal_emotional_coherence({"pleasure": 0.0, "arousal": 0.5}, None) is None
    )
    assert cross_modal_emotional_coherence({"arousal": 0.5}, lsm) is None
    assert cross_modal_emotional_coherence({"pleasure": 0.5}, lsm) is None
    assert (
        cross_modal_emotional_coherence({"pleasure": math.nan, "arousal": 0.5}, lsm)
        is None
    )
    assert (
        cross_modal_emotional_coherence(
            {"pleasure": 0.0, "arousal": 0.5}, {"intuitive_valence": 0.0}
        )
        is None
    )

    # MTC's PADState (pleasure in [-1, 1], arousal in [0, 1]) is read directly.
    pad_state = PADState(pleasure=0.5, arousal=0.75, dominance=0.5)
    assert cross_modal_emotional_coherence(
        pad_state, {"intuitive_valence": 0.5, "intuitive_arousal": 0.5}
    ) == pytest.approx(1.0)


def test_emotional_authenticity_from_real_signals():
    test = EmotionalAuthenticityTest(clock=_clock)
    state = {
        "pad_affect": {"pleasure": 0.4, "arousal": 0.7},
        "lsm_data_affect": {"intuitive_valence": 0.35, "intuitive_arousal": 0.45},
    }
    result = test.run(state)
    assert result.insufficient_data is False
    assert result.score > 0.7
    assert "affect_congruence" in result.components

    missing = test.run({"pad_affect": {"pleasure": 0.4, "arousal": 0.7}})
    assert missing.insufficient_data is True
    assert missing.score is None
    assert missing.passed is None

    unreal = test.run(
        {
            "pad_affect": {"pleasure": 0.4, "arousal": 0.7},
            "lsm_data": {
                "is_real": False,
                "intuitive_valence": 0.4,
                "intuitive_arousal": 0.4,
            },
        }
    )
    assert unreal.insufficient_data is True
    assert test.run({}).insufficient_data is True
    assert test.run(None).insufficient_data is True


def test_temporal_consistency_needs_samples():
    test = TemporalConsistencyTest(clock=_clock)
    pattern = [0.1, 0.5, 0.3, 0.9, 0.2, 0.7]

    def state(values, real=True):
        return {"snn_data": {"is_real": real, "temporal_patterns": list(values)}}

    assert test.run([state(pattern)]).insufficient_data is True
    assert test.run([]).insufficient_data is True
    assert test.run({"snn_data": {}}).insufficient_data is True
    assert test.run(None).insufficient_data is True
    assert test.run([state(pattern, False)] * 2).insufficient_data is True

    close = [0.12, 0.52, 0.31, 0.88, 0.21, 0.69]
    stable = test.run([state(pattern), state(close)])
    assert stable.insufficient_data is False
    assert "snn_temporal" in stable.components
    assert stable.score > 0.9
    assert stable.details["num_states"] == 2

    divergent = test.run([state(pattern), state([-v for v in pattern])])
    assert divergent.score == pytest.approx(0.0)

    constant = test.run([state([0.5] * 6), state([0.5] * 6)])
    assert constant.insufficient_data is True

    # Traits and beliefs score only when the states carry them.
    with_traits = test.run(
        [
            {"personality_traits": {"openness": 0.8, "warmth": 0.6}},
            {"personality_traits": {"openness": 0.7, "warmth": 0.6}},
        ]
    )
    assert with_traits.components == {"trait_stability": pytest.approx(0.95)}
    with_beliefs = test.run(
        [{"core_beliefs": ["a", "b"]}, {"core_beliefs": ["a", "c"]}]
    )
    assert with_beliefs.components == {"belief_consistency": pytest.approx(0.5)}


def test_temporal_consistency_reads_numpy_patterns():
    rng = np.random.default_rng(3)
    base = rng.random(30)
    states = [
        {"snn_data": {"is_real": True, "temporal_patterns": base}},
        {"snn_data": {"is_real": True, "temporal_patterns": base + 0.01}},
    ]
    result = TemporalConsistencyTest(clock=_clock).run(states)
    assert result.score == pytest.approx(1.0)


def test_conversation_continuity_from_word_overlap():
    assert contextual_continuity(None) is None
    assert contextual_continuity([]) is None
    assert contextual_continuity([{"role": "user", "content": "hello there"}]) is None

    turns = [
        {"role": "user", "content": "The garden needs water and light."},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": "Water the garden at dawn!"},
    ]
    # Previous words: the, garden, needs, water, and, light (6); shared: the,
    # garden, water (3); continuity = 3 / 6 * 2, clipped to 1.
    assert contextual_continuity(turns) == pytest.approx(1.0)

    sparse = [
        {"role": "system", "content": "one two three four five six seven eight"},
        {"role": "user", "content": "one zebra"},
    ]
    assert contextual_continuity(sparse) == pytest.approx(0.25)

    unrelated = [
        {"content": "alpha beta"},
        {"content": "gamma delta"},
    ]
    assert contextual_continuity(unrelated) == pytest.approx(0.0)

    # Word overlap alone never makes coherence measured: a reply that repeats
    # the previous turn would otherwise score 1.0 with no substrate at all.
    echo = [
        {"role": "user", "content": "tell me about cats"},
        {"role": "assistant", "content": "cats are about cats"},
    ]
    for turns in (sparse, echo):
        alone = CoherenceTest(clock=_clock).run({"conversation_history": turns})
        assert alone.insufficient_data is True
        assert alone.score is None
        assert alone.components == {}
        unreal = CoherenceTest(clock=_clock).run(
            {"snn_data": {"is_real": False, "synchrony": 0.9}}, turns
        )
        assert unreal.insufficient_data is True

    # Beside a real component, continuity joins the score.
    result = CoherenceTest(clock=_clock).run(
        {"snn_data": {"is_real": True, "synchrony": 0.25}}, sparse
    )
    assert result.components == {
        "snn": pytest.approx(0.25),
        "contextual": pytest.approx(0.25),
    }
    assert result.score == pytest.approx(0.25)
    assert result.passed is False
