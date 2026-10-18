"""The orchestrator's call of the legacy measurement framework: it passes the
substrate states, memories and thoughts only, and its log line after the
measurement accepts a metric that the framework could not compute (None)."""

import asyncio
import logging
from types import SimpleNamespace

import numpy as np

from mtc.neural.orchestrator import NeuralOrchestrator


def _orchestrator() -> NeuralOrchestrator:
    orchestrator = NeuralOrchestrator(emotion_bridge=False, memory_consolidator=False)
    orchestrator.experience_buffer = [
        SimpleNamespace(
            snn_result={"firing_rates": {"input": [0.1, 0.2], "output": [0.3]}},
            subconscious_state=SimpleNamespace(state_trajectory=np.zeros((3, 4))),
        )
    ]
    return orchestrator


class _Framework:
    """A stand-in with the framework's signature: no input_data parameter."""

    def __init__(self, metrics):
        self.metrics = metrics
        self.calls = []

    async def measure_consciousness(
        self, snn_states, lsm_states, htm_states=None, memories=None, thoughts=None
    ):
        self.calls.append(
            {
                "snn_states": snn_states,
                "lsm_states": lsm_states,
                "htm_states": htm_states,
                "memories": memories,
                "thoughts": thoughts,
            }
        )
        return self.metrics


def test_measurement_call_passes_no_input_data():
    metrics = SimpleNamespace(phi=0.25, integration=0.5, self_model_accuracy=0.75)
    framework = _Framework(metrics)
    orchestrator = _orchestrator()
    orchestrator.consciousness_framework = framework

    assert asyncio.run(orchestrator.measure_consciousness()) is metrics
    assert len(framework.calls) == 1
    call = framework.calls[0]
    assert call["htm_states"] is None
    assert call["memories"] == []
    np.testing.assert_array_equal(call["snn_states"], [0.1, 0.2, 0.3])
    assert np.asarray(call["lsm_states"]).shape == (3, 4)


def test_measurement_log_line_accepts_none_metrics(caplog):
    metrics = SimpleNamespace(phi=None, integration=None, self_model_accuracy=None)
    orchestrator = _orchestrator()
    orchestrator.consciousness_framework = _Framework(metrics)

    with caplog.at_level(logging.INFO, logger="mtc.neural.orchestrator"):
        result = asyncio.run(orchestrator.measure_consciousness())

    assert result is metrics
    lines = [r.getMessage() for r in caplog.records]
    measured = [line for line in lines if line.startswith("Consciousness measured")]
    assert measured == [
        "Consciousness measured: Phi=n/a, Integration=n/a, Self-Model=n/a"
    ]
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_measurement_log_line_formats_numbers():
    metrics = SimpleNamespace(phi=0.25, integration=None, self_model_accuracy=0.5)
    orchestrator = _orchestrator()
    orchestrator.consciousness_framework = _Framework(metrics)
    records = []

    class _Keep(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Keep(level=logging.INFO)
    log = logging.getLogger("mtc.neural.orchestrator")
    previous = log.level
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        assert asyncio.run(orchestrator.measure_consciousness()) is metrics
    finally:
        log.removeHandler(handler)
        log.setLevel(previous)
    assert "Consciousness measured: Phi=0.250, Integration=n/a, Self-Model=0.500" in (
        records
    )
