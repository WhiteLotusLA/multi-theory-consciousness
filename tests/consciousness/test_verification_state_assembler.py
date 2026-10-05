"""The state assembler hands the verifier only real substrate blocks, passes the
LSM's intuitive affect through when the LSM is real, and omits every
longitudinal input it was not given, so those tests read insufficient_data."""

from datetime import datetime, timezone

import pytest

from mtc.consciousness.pad_affect import PADState
from mtc.consciousness.verification_framework import ConsciousnessVerifier
from mtc.consciousness.verification_state_assembler import (
    ConsciousnessStateAssembler,
)

T0 = datetime(2000, 1, 1, tzinfo=timezone.utc)


def _neural(snn_real=True, lsm_real=True, htm_real=True):
    return {
        "snn": {"synchrony": 0.8, "temporal_patterns": [0.1, 0.2], "is_real": snn_real},
        "lsm": {
            "coherence": 0.7,
            "intuitive_valence": 0.4,
            "intuitive_arousal": -0.2,
            "is_real": lsm_real,
        },
        "htm": {"anomaly_score": 0.2, "is_real": htm_real},
        "memory": {"episodic_count": 3, "is_real": True},
    }


def test_assembler_gates_on_is_real():
    state = ConsciousnessStateAssembler.from_neural_data(
        _neural(snn_real=False, lsm_real=False, htm_real=False), now=T0
    )
    assert state["snn_data"] == {}
    assert state["lsm_data"] == {}
    assert state["htm_data"] == {}
    assert state["lsm_data_affect"] is None
    assert state["timestamp"] == T0.isoformat()

    real = ConsciousnessStateAssembler.from_neural_data(_neural(), now=T0)
    assert real["snn_data"]["synchrony"] == 0.8
    assert real["lsm_data"]["coherence"] == 0.7
    assert real["htm_data"]["anomaly_score"] == 0.2

    # Blocks without the flag, or with a truthy non-boolean flag, are not real.
    loose = {
        "snn": {"synchrony": 0.8},
        "lsm": {"coherence": 0.7, "is_real": "yes"},
        "htm": None,
    }
    state = ConsciousnessStateAssembler.from_neural_data(loose, now=T0)
    assert state["snn_data"] == {} and state["lsm_data"] == {}
    assert state["htm_data"] == {}
    assert ConsciousnessStateAssembler.from_neural_data(None, now=T0)["snn_data"] == {}


def test_omitted_longitudinal_inputs_yield_insufficient():
    state = ConsciousnessStateAssembler.from_neural_data(
        _neural(),
        conversation_history=[{"role": "user", "content": "hi"}],
        now=T0,
    )
    for key in ("temporal_data", "personality_traits", "core_beliefs"):
        assert key not in state
    assert state["pad_affect"] is None
    assert state["conversation_history"] == [{"role": "user", "content": "hi"}]

    suite = ConsciousnessVerifier().run_verification_suite(state)
    assert "temporal_consistency" in suite.insufficient
    assert "emotional_authenticity" in suite.insufficient
    assert suite.measured == ["coherence"]

    with_temporal = ConsciousnessStateAssembler.from_neural_data(
        _neural(), temporal_data={"states": [], "timestamps": []}, now=T0
    )
    assert with_temporal["temporal_data"] == {"states": [], "timestamps": []}


def test_lsm_affect_keys_passed_through():
    pad = PADState(pleasure=0.3, arousal=0.6, dominance=0.5)
    state = ConsciousnessStateAssembler.from_neural_data(
        _neural(), pad_state=pad, now=T0
    )
    assert state["lsm_data_affect"] == {
        "intuitive_valence": 0.4,
        "intuitive_arousal": -0.2,
    }
    assert state["pad_affect"] == {
        "pleasure": pytest.approx(0.3),
        "arousal": pytest.approx(0.6),
        "dominance": pytest.approx(0.5),
    }
    suite = ConsciousnessVerifier().run_verification_suite(state)
    assert "emotional_authenticity" in suite.measured

    mapping = ConsciousnessStateAssembler.from_neural_data(
        _neural(), pad_state={"pleasure": 0.3, "arousal": 0.6}, now=T0
    )
    assert mapping["pad_affect"] == {
        "pleasure": pytest.approx(0.3),
        "arousal": pytest.approx(0.6),
        "dominance": None,
    }

    unreal = ConsciousnessStateAssembler.from_neural_data(
        _neural(lsm_real=False), pad_state=pad, now=T0
    )
    assert unreal["lsm_data_affect"] is None


def test_inputs_are_not_mutated_or_shared():
    neural = _neural()
    state = ConsciousnessStateAssembler.from_neural_data(neural, now=T0)
    state["snn_data"]["synchrony"] = 0.0
    assert neural["snn"]["synchrony"] == 0.8


class _Integration:
    def __init__(self, neural, temporal):
        self._neural = neural
        self._temporal = temporal

    def collect_neural_data(self):
        return self._neural

    def get_temporal_data(self):
        return self._temporal


def test_from_integration_reads_collection_and_temporal_samples():
    temporal = {"states": [{"snn_data": {}}], "timestamps": [None]}
    integration = _Integration(_neural(), temporal)
    state = ConsciousnessStateAssembler.from_integration(
        integration, conversation_history=None, now=T0
    )
    assert state["snn_data"]["synchrony"] == 0.8
    assert state["temporal_data"] == temporal
    assert state["conversation_history"] == []

    assembler = ConsciousnessStateAssembler(integration, clock=lambda: T0)
    assert assembler.assemble() == state

    empty = ConsciousnessStateAssembler(None, clock=lambda: T0).assemble()
    assert empty["snn_data"] == {} and empty["htm_data"] == {}
    assert "temporal_data" not in empty

    class _Broken:
        def collect_neural_data(self):
            raise RuntimeError("collector down")

    broken = ConsciousnessStateAssembler.from_integration(_Broken(), now=T0)
    assert broken["snn_data"] == {} and broken["lsm_data_affect"] is None
