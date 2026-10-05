"""The merged emotion encoder: rate coding of ten channels into spikes.

Pure NumPy; a recording stand-in replaces the data manager, so no store client
is involved.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pytest

from mtc.neural.spiking import emotion_encoder
from mtc.neural.spiking.emotion_encoder import (
    EMOTION_CHANNELS,
    EmotionEncoder,
    EmotionState,
    OptimizedEmotionEncoder,
)
from mtc.neural.spiking.snn_core_optimized import OptimizedSpikingNeuralNetwork

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class RecordingManager:
    def __init__(self) -> None:
        self.batches: List[List[Any]] = []

    def store_spike_batch(self, spikes):
        self.batches.append(list(spikes))
        return [str(i) for i, _ in enumerate(spikes)]

    def status(self) -> Dict[str, Any]:
        return {"backend": "recording", "available": True, "last_error": None}


def _state(**emotions: float) -> EmotionState:
    return EmotionState(timestamp=T0, emotions=dict(emotions), valence=0.5, arousal=0.8)


def _snn(hidden: int = 30) -> OptimizedSpikingNeuralNetwork:
    return OptimizedSpikingNeuralNetwork(n_input=10, n_hidden=hidden, n_output=10)


def test_single_emotion_state_and_channel_list():
    from mtc.neural.liquid import emotion_lsm_bridge

    assert EMOTION_CHANNELS == (
        "joy",
        "sadness",
        "anger",
        "fear",
        "surprise",
        "disgust",
        "love",
        "trust",
        "anticipation",
        "curiosity",
    )
    assert EmotionEncoder.EMOTION_CHANNELS is EMOTION_CHANNELS
    assert OptimizedEmotionEncoder.EMOTION_CHANNELS is EMOTION_CHANNELS
    assert emotion_lsm_bridge.EmotionState is EmotionState
    assert EmotionState(T0, {}, 0.0, 0.0).dominance is None


def test_encoding_rate_codes_channels():
    encoder = EmotionEncoder(snn=_snn(), seed=1)
    result = encoder.encode_emotion(_state(joy=0.9), duration_ms=100.0)
    rates = result["firing_rates"]["input"]
    joy = EMOTION_CHANNELS.index("joy")
    assert rates[joy] > 0.0
    assert all(rate == 0.0 for i, rate in enumerate(rates) if i != joy)
    assert result["spikes_by_layer"]["input"] > 0
    assert result["total_spikes"] == sum(result["spikes_by_layer"].values())
    assert set(result["output_emotions"]) == set(EMOTION_CHANNELS)
    assert result["input_emotions"] == {"joy": 0.9}
    stronger = EmotionEncoder(snn=_snn(), seed=1).encode_emotion(
        _state(joy=0.9), 100.0
    )["firing_rates"]["input"][joy]
    weaker = EmotionEncoder(snn=_snn(), seed=1).encode_emotion(
        EmotionState(T0, {"joy": 0.2}, 0.0, 0.8), 100.0
    )["firing_rates"]["input"][joy]
    assert stronger >= weaker
    # counters are cleared per encoding, so a second call does not inflate rates
    again = encoder.encode_emotion(_state(joy=0.9), duration_ms=100.0)
    assert again["spikes_by_layer"]["input"] <= result["spikes_by_layer"]["input"] * 2


def test_unknown_channels_are_ignored_and_empty_state_is_silent():
    encoder = EmotionEncoder(snn=_snn(), seed=3)
    result = encoder.encode_emotion(_state(calm=0.9), duration_ms=50.0)
    assert result["spikes_by_layer"]["input"] == 0
    assert encoder.total_spikes_encoded == result["total_spikes"]


def test_neuron_ids_derived_from_layer_sizes():
    manager = RecordingManager()
    snn = _snn(hidden=150)
    encoder = EmotionEncoder(
        snn=snn,
        neural_data_manager=manager,
        store_spikes=True,
        max_stored_spikes=100000,
        seed=2,
    )
    encoder.encode_emotion(
        _state(joy=1.0, love=1.0, curiosity=1.0, fear=1.0), duration_ms=100.0
    )
    stored = [spike for batch in manager.batches for spike in batch]
    assert stored, "the drive is strong enough to produce spikes"
    by_layer: Dict[str, List[int]] = {"input": [], "hidden": [], "output": []}
    for spike in stored:
        by_layer[spike.metadata["layer"]].append(spike.neuron_id)
    assert all(0 <= n < 10 for n in by_layer["input"])
    assert all(10 <= n < 160 for n in by_layer["hidden"])
    assert all(160 <= n < 170 for n in by_layer["output"])
    # the fixed +100 and +200 offsets of the reference collided past 100 hidden
    all_ids = [n for ids in by_layer.values() for n in ids]
    assert len(set(all_ids)) <= 170
    assert stored[0].timestamp == T0


def test_spike_storage_off_by_default():
    manager = RecordingManager()
    for cls in (EmotionEncoder, OptimizedEmotionEncoder):
        encoder = cls(snn=_snn(), neural_data_manager=manager, seed=1)
        encoder.encode_emotion(_state(joy=1.0, fear=1.0), duration_ms=50.0)
        assert encoder.store_spikes is False
        assert encoder.status()["spike_storage"] is False
    assert manager.batches == []
    with pytest.raises(ValueError):
        EmotionEncoder(store_spikes=True)
    with pytest.raises(ValueError):
        OptimizedEmotionEncoder(store_spikes=True)


def test_stored_spike_count_is_bounded():
    manager = RecordingManager()
    encoder = EmotionEncoder(
        snn=_snn(),
        neural_data_manager=manager,
        store_spikes=True,
        max_stored_spikes=5,
        seed=4,
    )
    encoder.encode_emotion(_state(joy=1.0, love=1.0, fear=1.0), duration_ms=100.0)
    assert len(manager.batches[0]) == 5


def test_stored_membrane_potential_is_measured_or_none():
    manager = RecordingManager()
    snn = _snn()
    encoder = EmotionEncoder(
        snn=snn, neural_data_manager=manager, store_spikes=True, seed=5
    )
    encoder.encode_emotion(_state(joy=1.0, love=1.0), duration_ms=100.0)
    stored = [spike for batch in manager.batches for spike in batch]
    assert stored
    for spike in stored:
        potential = spike.membrane_potential
        assert potential is not None
        assert potential != -55.0 or spike.metadata["layer"] == "never"
        assert -100.0 < potential < 0.0

    class Bare:
        """A network that exposes no membrane potentials."""

        dt = 1.0
        current_time = 0.0

        def simulate_step(self, currents):
            self.current_time += 1.0
            return {
                "input": np.ones(10, dtype=bool),
                "hidden": np.zeros(4, dtype=bool),
                "output": np.zeros(10, dtype=bool),
            }

        def get_firing_rates(self, window=100.0):
            return {
                k: np.zeros(n)
                for k, n in (("input", 10), ("hidden", 4), ("output", 10))
            }

    manager2 = RecordingManager()
    bare = EmotionEncoder(
        snn=Bare(), neural_data_manager=manager2, store_spikes=True, seed=1
    )
    bare.encode_emotion(_state(joy=1.0), duration_ms=3.0)
    assert manager2.batches[0]
    assert all(s.membrane_potential is None for s in manager2.batches[0])
    ids = [s.neuron_id for s in manager2.batches[0]]
    assert max(ids) == 9  # only input neurons spiked; offsets start at zero


def test_optimized_process_emotion_derives_valence_and_arousal(monkeypatch):
    encoder = OptimizedEmotionEncoder(snn=_snn(), seed=1)
    captured: Dict[str, EmotionState] = {}
    original = encoder.encode_emotion

    def spy(state, duration_ms=100.0):
        captured["state"] = state
        return original(state, duration_ms)

    monkeypatch.setattr(encoder, "encode_emotion", spy)
    result = encoder.process_emotion({"joy": 0.9, "sadness": 0.3}, context="ignored")
    state = captured["state"]
    assert state.valence == pytest.approx(0.9 / 3 - 0.3 / 3)
    assert state.arousal == pytest.approx((0.9 + 0.3) / 2)
    assert state.dominance is None
    assert state.timestamp.tzinfo is not None
    assert result["total_spikes"] >= 0
    encoder.process_emotion({})
    assert captured["state"].arousal == 0.0 and captured["state"].valence == 0.0


def test_valence_word_sets_are_documented_defaults_and_overridable(monkeypatch):
    assert emotion_encoder.DEFAULT_POSITIVE_EMOTIONS == ("joy", "love", "trust")
    assert emotion_encoder.DEFAULT_NEGATIVE_EMOTIONS == ("sadness", "anger", "fear")
    encoder = EmotionEncoder(
        snn=_snn(),
        positive_emotions=("curiosity",),
        negative_emotions=("disgust",),
        seed=1,
    )
    seen: Dict[str, EmotionState] = {}
    original = encoder.encode_emotion

    def spy(state, duration_ms=100.0):
        seen["state"] = state
        return original(state, duration_ms)

    monkeypatch.setattr(encoder, "encode_emotion", spy)
    encoder.process_emotion({"curiosity": 0.6, "joy": 0.9})
    assert seen["state"].valence == pytest.approx(0.6)


def test_summary_record_has_no_sentinel_neuron():
    manager = RecordingManager()
    encoder = OptimizedEmotionEncoder(
        snn=_snn(), neural_data_manager=manager, store_spikes=True, seed=1
    )
    encoder.encode_emotion(_state(joy=0.8, fear=0.4), duration_ms=50.0)
    assert len(manager.batches) == 1 and len(manager.batches[0]) == 1
    summary = manager.batches[0][0]
    assert summary.neuron_id is None  # no -1 sentinel
    assert summary.membrane_potential is None
    assert summary.metadata["record"] == "summary"
    assert summary.emotion_type == "fear,joy"
    assert summary.intensity == pytest.approx(0.6)
    assert summary.timestamp == T0
    for key in ("total_spikes", "avg_input_rate", "avg_hidden_rate", "avg_output_rate"):
        assert key in summary.metadata


def test_naive_timestamp_is_read_as_local_time_and_stored_in_utc():
    manager = RecordingManager()
    encoder = OptimizedEmotionEncoder(
        snn=_snn(), neural_data_manager=manager, store_spikes=True, seed=1
    )
    naive = datetime(2000, 6, 1, 9, 0, 0)
    encoder.encode_emotion(
        EmotionState(naive, {"joy": 0.5}, 0.0, 0.5), duration_ms=10.0
    )
    stamp = manager.batches[0][0].timestamp
    assert stamp == naive.astimezone(timezone.utc) and stamp.tzinfo is not None


def test_global_random_state_is_left_alone():
    first, second = EmotionEncoder(snn=_snn(), seed=None), OptimizedEmotionEncoder(
        snn=_snn(), seed=None
    )
    np.random.seed(1234)
    expected = np.random.random()
    np.random.seed(1234)
    first.encode_emotion(_state(joy=1.0), 20.0)
    second.encode_emotion(_state(joy=1.0), 20.0)
    assert np.random.random() == expected


def test_seed_makes_encoding_reproducible():
    first = EmotionEncoder(snn=_snn(), seed=9)
    second = EmotionEncoder(snn=_snn(), seed=9)
    for encoder in (first, second):
        encoder.snn.W_input_hidden[:] = 120.0
        encoder.snn.W_hidden_output[:] = 100.0
    a = first.encode_emotion(_state(joy=0.7), 50.0)["firing_rates"]["input"]
    b = second.encode_emotion(_state(joy=0.7), 50.0)["firing_rates"]["input"]
    assert a == b


def test_storage_failures_are_reported_not_raised():
    class Failing(RecordingManager):
        def store_spike_batch(self, spikes):
            raise RuntimeError("driver rejected the batch")

    encoder = OptimizedEmotionEncoder(
        snn=_snn(), neural_data_manager=Failing(), store_spikes=True, seed=1
    )
    result = encoder.encode_emotion(_state(joy=0.8), duration_ms=20.0)
    assert result["total_spikes"] >= 0
    assert "driver rejected" in encoder.status()["last_error"]


def test_pattern_similarity_uses_recorded_events_not_the_wall_clock():
    encoder = EmotionEncoder(snn=_snn(), seed=1)
    assert encoder.get_emotion_pattern_similarity("joy", "fear") == 0.0
    encoder.encode_emotion(_state(joy=1.0, fear=1.0), duration_ms=100.0)
    same = encoder.get_emotion_pattern_similarity("joy", "joy")
    assert same in (0.0, 1.0)
    assert 0.0 <= encoder.get_emotion_pattern_similarity("joy", "fear") <= 1.0
    assert encoder.get_emotion_pattern_similarity("joy", "calm") == 0.0


def test_status_and_metrics_shape():
    encoder = OptimizedEmotionEncoder(snn=_snn(), seed=1)
    status = encoder.status()
    assert status["backend"] == "numpy" and status["available"] is True
    assert status["last_error"] is None and status["channels"] == 10
    encoder.encode_emotion(_state(joy=1.0), 20.0)
    metrics = encoder.get_metrics()
    assert metrics["emotions_tracked"] == list(EMOTION_CHANNELS)
    assert metrics["spike_storage_enabled"] is False
    assert metrics["snn_metrics"]["total_neurons"] == 50


def test_module_import_loads_no_store_driver():
    import subprocess
    import sys

    code = (
        "import sys, mtc.neural.spiking.emotion_encoder;"
        "bad=[m for m in ('pymongo','motor') if m in sys.modules];"
        "sys.exit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-1000:]
