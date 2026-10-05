"""Oscillatory binding runs on every experience from measured inputs only, and
the substrate's latest outputs are stashed for the measurement layer without
inventing the ones that were not produced. Substrates are fakes."""

from types import SimpleNamespace

import numpy as np
import pytest

import mtc.neural.orchestrator as orch_mod
from mtc.neural.orchestrator import NeuralOrchestrator
from mtc.neural.oscillatory_binding import OSCILLATOR_NAMES, BindingResult

LSM_NAMES = OSCILLATOR_NAMES[8:16]
HTM_NAMES = OSCILLATOR_NAMES[16:24]
SNN_OUTPUT = np.array([0.2, 0.8, 0.1, 0.5])
TRAJECTORY = np.linspace(0.0, 1.0, 32).reshape(8, 4)
EPISODIC_PATTERN = np.linspace(-1.0, 1.0, 6)


class _FakeHTM:
    def __init__(self):
        self.input_size = 16
        self.calls = []
        self.config = SimpleNamespace(
            num_columns=64, cells_per_column=4, local_area_density=0.05
        )

    def process(self, pattern, learn=True):
        self.calls.append((np.asarray(pattern).shape, learn))
        return {"anomaly_score": 0.25, "num_active_columns": 3}


class _FakeBridge:
    def __init__(self, **kwargs):
        self.snn = SimpleNamespace(_count_neurons=lambda: 40)
        self.lsm = SimpleNamespace(
            config=SimpleNamespace(reservoir_size=50, spectral_radius=0.9)
        )

    def process_emotion(self, emotion_state, snn_duration_ms, lsm_duration_ms):
        snn_result = {
            "firing_rates": {"output": SNN_OUTPUT.copy()},
            "metrics": {"synchrony": 0.4},
            "total_spikes": 11,
        }
        subconscious = SimpleNamespace(
            state_trajectory=TRAJECTORY,
            emotional_coherence=0.6,
            dominant_pattern="calm",
        )
        return snn_result, subconscious


class _FakeConsolidator:
    def __init__(self, **kwargs):
        self.htm = _FakeHTM()
        self.episodic_memories = []
        self.semantic_patterns = []

    def add_episodic_memory(self, **kwargs):
        memory = SimpleNamespace(
            subconscious_pattern=EPISODIC_PATTERN, consolidation_strength=0.0
        )
        self.episodic_memories.append(memory)
        return memory

    def consolidate_memories(self, memories, sleep_duration):
        return {"patterns_discovered": 0}


@pytest.fixture
def wired(monkeypatch):
    monkeypatch.setattr(orch_mod, "_HAS_EMOTION_BRIDGE", True)
    monkeypatch.setattr(orch_mod, "EmotionLSMBridge", _FakeBridge)
    monkeypatch.setattr(orch_mod, "EmotionState", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(orch_mod, "_HAS_MEMORY_CONSOLIDATOR", True)
    monkeypatch.setattr(orch_mod, "MemoryConsolidator", _FakeConsolidator)
    return NeuralOrchestrator(enable_mongodb=False)


def test_process_experience_runs_binding(wired):
    assert wired._last_binding_result is None
    wired.process_experience("observation", {"joy": 0.7, "curiosity": 0.4})

    result = wired._last_binding_result
    assert isinstance(result, BindingResult)
    assert 0.0 <= result.global_order_parameter <= 1.0 + 1e-9
    # The live HTM pass is inference only.
    assert wired.memory_consolidator.htm.calls
    assert all(learn is False for _, learn in wired.memory_consolidator.htm.calls)

    wired.process_experience("observation", {"joy": 0.2})
    assert wired.get_binding_stats()["binding_count"] == 2


def test_binding_stats_report_latest_values(wired):
    assert wired.get_binding_stats() == {}
    wired.process_experience("observation", {"joy": 0.7})
    result = wired._last_binding_result
    assert wired.get_binding_stats() == {
        "global_order_parameter": result.global_order_parameter,
        "cross_coherence": result.cross_coherence,
        "binding_count": 1,
    }


def test_binding_context_string(wired):
    assert wired.get_binding_context() == ""
    wired.process_experience("observation", {"joy": 0.7})
    context = wired.get_binding_context()
    assert isinstance(context, str) and context
    assert context == wired._last_binding_result.context_string


def test_unmeasured_cross_modal_metrics_are_left_unset(wired):
    subconscious = SimpleNamespace(state_trajectory=TRAJECTORY, emotional_coherence=0.6)
    episodic = SimpleNamespace(consolidation_strength=0.0)
    metrics = wired._extract_binding_metrics(
        subconscious,
        episodic,
        emotions={"joy": 0.5, "surprise": 0.3},
        valence=0.2,
        arousal=0.6,
    )
    for name in (
        "attention_focus",
        "goal_relevance",
        "temporal_urgency",
        "input_sensitivity",
    ):
        assert name not in metrics
    # Emotions that were not reported stay unset as well.
    assert "fear" not in metrics and "calm" not in metrics
    assert metrics["joy"] == pytest.approx(0.5)
    assert metrics["novelty"] == pytest.approx(0.3)
    assert metrics["valence"] == pytest.approx(0.6)
    assert metrics["arousal"] == pytest.approx(0.6)
    assert metrics["fading_memory"] == pytest.approx(0.6)
    assert metrics["anomaly_score"] == pytest.approx(0.25)
    assert set(metrics) <= set(OSCILLATOR_NAMES)
    assert all(np.isfinite(value) for value in metrics.values())


def test_latest_signals_none_until_first_experience(wired):
    assert wired.latest_neural_signals is None
    assert wired.latest_substrate_states is None
    assert wired._last_retrieved_memories == []

    wired.process_experience("observation", {"joy": 0.7})

    signals = wired.latest_neural_signals
    assert set(signals) == {"snn", "lsm", "htm"}
    for value in signals.values():
        assert isinstance(value, np.ndarray)
        assert value.ndim == 1 and value.dtype == float
    assert np.array_equal(signals["snn"], SNN_OUTPUT)
    assert np.array_equal(signals["lsm"], TRAJECTORY[-1])
    assert np.array_equal(signals["htm"], EPISODIC_PATTERN)

    states = wired.latest_substrate_states
    assert states["snn_state"]["total_spikes"] == 11
    assert states["lsm_state"]["reservoir_state"] is TRAJECTORY
    assert states["lsm_state"]["spectral_radius"] == pytest.approx(0.9)


def test_missing_substrate_outputs_stay_missing():
    bare = NeuralOrchestrator(enable_mongodb=False)
    bare.process_experience("observation", {"joy": 0.7})

    assert bare.latest_neural_signals is None
    assert bare.latest_substrate_states is None
    # Binding still ran on the emotions it was given.
    assert isinstance(bare._last_binding_result, BindingResult)
    metrics = bare._extract_binding_metrics(
        None, None, emotions={"joy": 0.7}, valence=0.1, arousal=0.5
    )
    assert not set(metrics) & set(LSM_NAMES)
    assert not set(metrics) & set(HTM_NAMES)

    bare._stash_latest_signals({"firing_rates": {"output": [0.3, 0.1]}}, None, None)
    assert set(bare.latest_neural_signals) == {"snn"}
    assert set(bare.latest_substrate_states) == {"snn_state"}

    bare._stash_latest_signals(None, None, None)
    assert bare.latest_neural_signals is None
    assert bare.latest_substrate_states is None
