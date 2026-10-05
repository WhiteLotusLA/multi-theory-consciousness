"""The emotion to reservoir bridge on small configs (hidden 32 by 32, reservoir 50).

No store is involved: a recording stand-in replaces the data manager. The
production path tests skip without torch, snntorch and reservoirpy.
"""

import sys
from datetime import datetime, timezone
from typing import Any, List

import numpy as np
import pytest

from mtc.core import config as config_module
from mtc.core.config import Settings
from mtc.neural.liquid.emotion_lsm_bridge import (
    EmotionLSMBridge,
    EmotionLSMBridgeConfig,
    EmotionState,
    SubconsciousState,
)

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _config(**changes: Any) -> EmotionLSMBridgeConfig:
    values = dict(
        snn_hidden_layers=(32, 32),
        reservoir_size=50,
        lsm_input_dim=20,
        lsm_output_dim=6,
        numpy_hidden_neurons=20,
        numpy_reservoir_size=40,
        device="cpu",
        seed=1,
    )
    values.update(changes)
    return EmotionLSMBridgeConfig(**values)


def _state(stamp: datetime = T0, **emotions: float) -> EmotionState:
    return EmotionState(
        timestamp=stamp,
        emotions=emotions or {"joy": 0.9, "fear": 0.2},
        valence=0.4,
        arousal=0.6,
    )


class RecordingManager:
    def __init__(self) -> None:
        self.liquid_states: List[Any] = []

    def store_liquid_state(self, state):
        self.liquid_states.append(state)
        return "id"

    def status(self):
        return {"backend": "recording", "available": True, "last_error": None}


@pytest.fixture
def production_libraries():
    pytest.importorskip("torch")
    pytest.importorskip("snntorch")
    pytest.importorskip("reservoirpy")


@pytest.fixture
def numpy_bridge() -> EmotionLSMBridge:
    return EmotionLSMBridge(config=_config(production=False))


@pytest.fixture
def production_bridge(production_libraries) -> EmotionLSMBridge:
    return EmotionLSMBridge(config=_config())


def test_production_path_when_neural_extra_present(production_bridge):
    bridge = production_bridge
    assert bridge.path == "production"
    status = bridge.status()
    assert status["backend"] == "production" and status["available"] is True
    assert status["reason"] is None and status["device"] == "cpu"
    snn_result, state = bridge.process_emotion(_state())
    assert isinstance(state, SubconsciousState) and state.path == "production"
    assert len(snn_result["firing_rates"]["output"]) == 10
    assert snn_result["output_unit"] == "spike_fraction"
    assert isinstance(snn_result["total_spikes"], int)
    assert snn_result["metrics"]["synchrony"] is not None
    assert bridge.get_metrics()["snn_metrics"]["production"] is True
    assert bridge.snn._count_neurons() == 10 + 32 + 32 + 10


def test_numpy_path_otherwise_and_status_says_so(
    production_libraries, monkeypatch, caplog
):
    monkeypatch.setitem(sys.modules, "reservoirpy", None)
    with caplog.at_level("WARNING", logger="mtc.neural.liquid.emotion_lsm_bridge"):
        bridge = EmotionLSMBridge(config=_config())
    assert bridge.path == "numpy"
    status = bridge.status()
    assert status["backend"] == "numpy" and "reservoirpy" in status["reason"]
    assert any("NumPy path" in r.getMessage() for r in caplog.records)
    snn_result, state = bridge.process_emotion(_state())
    assert state.path == "numpy"
    assert snn_result["output_unit"] == "hz"
    assert bridge.get_metrics()["snn_metrics"]["production"] is False

    with pytest.raises(ImportError) as caught:
        EmotionLSMBridge(config=_config(production=True))
    assert "neural" in str(caught.value)

    forced = EmotionLSMBridge(config=_config(production=False))
    assert forced.status()["reason"] == "production=False in the config"


def test_calls_process_input(production_bridge, monkeypatch):
    bridge = production_bridge
    seen = []
    original = bridge.lsm.process_input

    def spy(data):
        seen.append(np.asarray(data).shape)
        return original(data)

    monkeypatch.setattr(bridge.lsm, "process_input", spy)
    bridge.process_emotion(_state(), snn_duration_ms=50.0, lsm_duration_ms=100.0)
    assert len(seen) == bridge.config.lsm_chunks
    assert all(shape[1] == 20 for shape in seen)
    assert sum(shape[0] for shape in seen) == 100  # one input row per millisecond


def test_snn_device_from_setting(production_libraries, monkeypatch):
    monkeypatch.setattr(config_module, "_settings", Settings(DEVICE="cpu"))
    bridge = EmotionLSMBridge(config=_config(device=None))
    assert str(bridge.snn.device) == "cpu"
    assert bridge.status()["device"] == "cpu"
    # an explicit device in the config wins over a setting that would raise
    monkeypatch.setattr(config_module, "_settings", Settings(DEVICE="cuda"))
    bridge = EmotionLSMBridge(config=_config(device="cpu"))
    assert str(bridge.snn.device) == "cpu"


def test_untrained_readout_reports_none_not_values(numpy_bridge, production_bridge):
    for bridge in (numpy_bridge, production_bridge):
        _, state = bridge.process_emotion(_state())
        assert state.readout_trained is False
        assert state.intuitive_valence is None
        assert state.intuitive_arousal is None
        assert state.emotional_coherence is None
        # measured from the reservoir, so still reported
        assert isinstance(state.reservoir_energy, float)
        assert state.reservoir_entropy is not None
        assert isinstance(state.reservoir_sparsity, float)
        assert isinstance(state.dominant_pattern, str)
        assert state.state_trajectory and all(
            isinstance(v, float) for v in state.state_trajectory
        )
        summary = bridge.get_intuitive_summary()
        assert summary["readout_trained"] is False
        assert summary["intuitive_feelings"] == {
            "valence": None,
            "arousal": None,
            "coherence": None,
            "valence_trend": None,
        }
        assert bridge.status()["readout_trained"] is False


def test_trained_readout_reports_values_numpy(numpy_bridge):
    rng = np.random.default_rng(0)
    inputs = [rng.random((130, 10)) for _ in range(2)]
    targets = [np.tile([0.5, -0.25, 0.75, 0.0, 0.0, 0.0], (130, 1)) for _ in range(2)]
    numpy_bridge.train_readout(inputs, targets)
    assert numpy_bridge.readout_trained() is True
    _, state = numpy_bridge.process_emotion(_state())
    assert state.readout_trained is True
    for value in (
        state.intuitive_valence,
        state.intuitive_arousal,
        state.emotional_coherence,
    ):
        assert isinstance(value, float) and np.isfinite(value)
    summary = numpy_bridge.get_intuitive_summary()
    assert summary["intuitive_feelings"]["valence"] is not None


def test_trained_readout_reports_values_production(production_bridge):
    bridge = production_bridge
    rng = np.random.default_rng(0)
    inputs = rng.random((2, 20, 20))
    targets = np.tile([0.5, -0.25, 0.75, 0.0, 0.0, 0.0], (2, 20, 1))
    assert bridge.readout_trained() is False
    bridge.train_readout(inputs, targets)
    assert bridge.readout_trained() is True
    _, state = bridge.process_emotion(_state())
    assert state.readout_trained is True
    assert all(
        isinstance(v, float)
        for v in (
            state.intuitive_valence,
            state.intuitive_arousal,
            state.emotional_coherence,
        )
    )


def test_pattern_thresholds_from_config(numpy_bridge):
    flat = [1.0] * 12
    assert numpy_bridge._classify_pattern(flat)[0] == "stable"
    wild = [0.0, 100.0] * 6
    assert numpy_bridge._classify_pattern(wild)[0] == "chaotic"
    short = numpy_bridge._classify_pattern([1.0, 2.0])
    assert short == ("insufficient_data", None)
    lenient = EmotionLSMBridge(
        config=_config(
            production=False,
            chaotic_energy_std=1000.0,
            stable_energy_std=0.1,
            rhythmic_sign_changes=5,
        )
    )
    assert lenient._classify_pattern(wild)[0] == "rhythmic"
    stricter = EmotionLSMBridge(
        config=_config(production=False, min_states_for_pattern=20)
    )
    assert stricter._classify_pattern(flat)[0] == "insufficient_data"
    assert numpy_bridge._prediction_confidence([1.0, 2.0]) is None


def test_transitional_strength_is_none_not_a_constant(numpy_bridge):
    """The residual class has no measured strength; the other classes do."""
    cfg = numpy_bridge.config
    # a spread strictly between the stable and chaotic thresholds, no rhythm
    middle = (cfg.stable_energy_std + cfg.chaotic_energy_std) / 2.0
    ramp = [middle * 0.2 * index for index in range(12)]
    assert cfg.stable_energy_std < float(np.std(ramp)) < cfg.chaotic_energy_std
    assert numpy_bridge._classify_pattern(ramp) == ("transitional", None)
    for energies, name in (([1.0] * 12, "stable"), ([0.0, 100.0] * 6, "chaotic")):
        pattern, strength = numpy_bridge._classify_pattern(energies)
        assert pattern == name and isinstance(strength, float)
        assert 0.0 <= strength <= 1.0


def test_history_bounded(numpy_bridge):
    bridge = EmotionLSMBridge(config=_config(production=False, history_limit=3))
    for _ in range(5):
        bridge.process_emotion(_state(), snn_duration_ms=10.0, lsm_duration_ms=12.0)
    assert len(bridge.state_history) == 3
    assert len(bridge.reservoir_trajectory) == 3
    assert bridge.states_processed == 5


def test_liquid_state_storage_only_when_enabled():
    manager = RecordingManager()
    quiet = EmotionLSMBridge(
        neural_data_manager=manager,
        store_states=False,
        config=_config(production=False),
    )
    quiet.process_emotion(_state(), lsm_duration_ms=12.0)
    assert manager.liquid_states == []
    storing = EmotionLSMBridge(
        neural_data_manager=manager, store_states=True, config=_config(production=False)
    )
    storing.process_emotion(_state(), lsm_duration_ms=12.0)
    assert len(manager.liquid_states) == 1
    stored = manager.liquid_states[0]
    assert stored.readout_values is None
    assert stored.coherence is None and stored.emotional_valence is None
    assert stored.arousal_level is None
    assert stored.reservoir_size == 40 and len(stored.neuron_states) == 40
    assert stored.timestamp == T0 and len(stored.input_pattern) == 10
    assert stored.metadata["readout_trained"] is False
    with pytest.raises(ValueError):
        EmotionLSMBridge(store_states=True, config=_config(production=False))


def test_storage_errors_are_reported_not_raised():
    class Failing(RecordingManager):
        def store_liquid_state(self, state):
            raise RuntimeError("write rejected")

    bridge = EmotionLSMBridge(
        neural_data_manager=Failing(),
        store_states=True,
        config=_config(production=False),
    )
    bridge.process_emotion(_state(), lsm_duration_ms=12.0)
    assert "write rejected" in bridge.status()["last_error"]


def test_state_dict_round_trip(numpy_bridge):
    for _ in range(3):
        numpy_bridge.process_emotion(
            _state(), snn_duration_ms=10.0, lsm_duration_ms=12.0
        )
    saved = numpy_bridge.to_state_dict()
    assert set(saved) == {"scalars", "arrays", "histories"}
    fresh = EmotionLSMBridge(config=_config(production=False))
    fresh.from_state_dict(saved)
    assert fresh.states_processed == 3
    assert len(fresh.state_history) == 3
    assert fresh.state_history[0].dominant_pattern == (
        numpy_bridge.state_history[0].dominant_pattern
    )
    assert fresh.state_history[-1].state_trajectory == (
        numpy_bridge.state_history[-1].state_trajectory
    )
    assert fresh.to_state_dict()["histories"] == saved["histories"]
    # the restored bridge does not claim a trained readout
    assert fresh.readout_trained() is False


def test_restore_skips_malformed_entries_and_never_raises(numpy_bridge, caplog):
    numpy_bridge.process_emotion(_state(), snn_duration_ms=10.0, lsm_duration_ms=12.0)
    good = numpy_bridge.to_state_dict()
    entries = good["histories"]["subconscious_states"]
    bad = [
        "not a dict",
        {"timestamp": "nonsense"},
        {**entries[0], "reservoir_energy": float("nan")},
        entries[0],
    ]
    fresh = EmotionLSMBridge(config=_config(production=False))
    with caplog.at_level("WARNING"):
        fresh.from_state_dict(
            {
                "scalars": {"states_processed": 7},
                "histories": {"subconscious_states": bad},
            }
        )
    assert len(fresh.state_history) == 1 and fresh.states_processed == 7
    assert any("skipped 3" in r.getMessage() for r in caplog.records)
    fresh.from_state_dict({"histories": {"subconscious_states": "garbage"}})
    fresh.from_state_dict({})
    assert len(fresh.state_history) == 1


def test_orchestrator_call_signature(numpy_bridge):
    naive = datetime(2000, 1, 1, 12, 0, 0)  # the orchestrator stamps local time
    snn_result, state = numpy_bridge.process_emotion(
        _state(stamp=naive), snn_duration_ms=50.0, lsm_duration_ms=100.0
    )
    assert state.timestamp == naive
    assert state.metadata["lsm_steps"] == 100
    for key in ("firing_rates", "total_spikes", "processing_time_ms"):
        assert key in snn_result
    # the integration reads these fields by name
    for name in (
        "emotional_coherence",
        "dominant_pattern",
        "intuitive_valence",
        "intuitive_arousal",
        "state_trajectory",
    ):
        assert hasattr(state, name)
    assert isinstance(snn_result["firing_rates"]["output"], list)


def test_sizes_for_the_orchestrator_on_both_paths(numpy_bridge, production_bridge):
    assert production_bridge.lsm.config.reservoir_size == 50
    assert production_bridge.snn._count_neurons() == 84
    assert numpy_bridge.lsm.n_reservoir == 40
    assert numpy_bridge.snn.get_metrics()["total_neurons"] == 40


def test_pattern_class_needs_enough_states_on_the_production_path(production_bridge):
    _, state = production_bridge.process_emotion(_state())
    assert len(state.state_trajectory) == production_bridge.config.lsm_chunks
    assert state.dominant_pattern != "insufficient_data"


def test_global_random_state_is_left_alone(numpy_bridge):
    np.random.seed(77)
    expected = np.random.random()
    np.random.seed(77)
    numpy_bridge.process_emotion(_state(), snn_duration_ms=10.0, lsm_duration_ms=12.0)
    assert np.random.random() == expected


def test_silent_reservoir_has_no_entropy(numpy_bridge):
    from mtc.neural.liquid.emotion_lsm_bridge import _normalized_entropy

    assert _normalized_entropy(np.zeros(10)) is None
    assert _normalized_entropy(np.array([1.0])) is None
    even = _normalized_entropy(np.ones(8))
    assert even == pytest.approx(1.0)
    peaked = _normalized_entropy(np.array([1.0, 0.0, 0.0, 0.0]))
    assert peaked == pytest.approx(0.0)
